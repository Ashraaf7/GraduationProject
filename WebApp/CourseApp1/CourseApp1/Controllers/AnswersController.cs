using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using CourseApp1.data;
using Microsoft.AspNet.Identity;

namespace CourseApp1.Controllers
{
    public class AnswersController : Controller
    {
        private courses_dbEntities db = new courses_dbEntities();

        // GET: Answers
        public ActionResult Index(int? id)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            var answers = db.Answers.Include(a => a.RoomQuestion).Include(a => a.AspNetUser).Where(t=>t.QuestionId==id);
            ViewBag.Question = id;
            return View(answers.ToList());
        }

        // GET: Answers/Create
        [Authorize]
        public ActionResult Create()
        {
            ViewBag.QuestionId = new SelectList(db.RoomQuestions, "QuestionId", "Content");
            ViewBag.UserId = new SelectList(db.AspNetUsers, "Id", "Email");
            return View();
        }

        // POST: Answers/Create
        // To protect from overposting attacks, enable the specific properties you want to bind to, for 
        // more details see https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "AnswerId,Content,QuestionId,UserId")] Answer answer,int? id)
        {
            answer.QuestionId = (int)id;
            answer.UserId = User.Identity.GetUserId();
            answer.CreationDate= DateTime.Now;
            if (ModelState.IsValid)
            {
                db.Answers.Add(answer);
                db.SaveChanges();
                return RedirectToAction("Index", new {id=id});
            }

            ViewBag.QuestionId = new SelectList(db.RoomQuestions, "QuestionId", "Content", answer.QuestionId);
            ViewBag.UserId = new SelectList(db.AspNetUsers, "Id", "Email", answer.UserId);
            return View(answer);
        }

        //// GET: Answers/Edit/5
        //public ActionResult Edit(int? id)
        //{
        //    if (id == null)
        //    {
        //        return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
        //    }
        //    Answer answer = db.Answers.Find(id);
        //    if (answer == null)
        //    {
        //        return HttpNotFound();
        //    }
        //    ViewBag.QuestionId = new SelectList(db.RoomQuestions, "QuestionId", "Content", answer.QuestionId);
        //    ViewBag.UserId = new SelectList(db.AspNetUsers, "Id", "Email", answer.UserId);
        //    return View(answer);
        //}

        //// POST: Answers/Edit/5
        //// To protect from overposting attacks, enable the specific properties you want to bind to, for 
        //// more details see https://go.microsoft.com/fwlink/?LinkId=317598.
        //[HttpPost]
        //[ValidateAntiForgeryToken]
        //public ActionResult Edit([Bind(Include = "AnswerId,Content,QuestionId,UserId")] Answer answer)
        //{
        //    if (ModelState.IsValid)
        //    {
        //        db.Entry(answer).State = EntityState.Modified;
        //        db.SaveChanges();
        //        return RedirectToAction("Index");
        //    }
        //    ViewBag.QuestionId = new SelectList(db.RoomQuestions, "QuestionId", "Content", answer.QuestionId);
        //    ViewBag.UserId = new SelectList(db.AspNetUsers, "Id", "Email", answer.UserId);
        //    return View(answer);
        //}

        //// GET: Answers/Delete/5
        //public ActionResult Delete(int? id)
        //{
        //    if (id == null)
        //    {
        //        return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
        //    }
        //    Answer answer = db.Answers.Find(id);
        //    if (answer == null)
        //    {
        //        return HttpNotFound();
        //    }
        //    return View(answer);
        //}

        //// POST: Answers/Delete/5
        //[HttpPost, ActionName("Delete")]
        //[ValidateAntiForgeryToken]
        //public ActionResult DeleteConfirmed(int id)
        //{
        //    Answer answer = db.Answers.Find(id);
        //    db.Answers.Remove(answer);
        //    db.SaveChanges();
        //    return RedirectToAction("Index");
        //}

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                db.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
