using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using CourseApp1.data;

namespace CourseApp1.Areas.Admin.Controllers
{
    [Authorize(Roles = "Admin")]
    public class Roadmap_courses1Controller : Controller
    {
        private courses_dbEntities db = new courses_dbEntities();
        // GET: Admin/Roadmap_courses1
        public ActionResult Index(int? id)
        {
            var roadmap_courses = db.Roadmap_courses.Include(r => r.Roadmap);
            roadmap_courses = roadmap_courses.Where(c => c.RoadmapId == id).OrderBy(c => c.Order);
            ViewBag.RoadmapId = id;
            return View(roadmap_courses.ToList());
        }
        // GET: Admin/Roadmap_courses/Details/5
        public ActionResult Details(int? id)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            Roadmap_courses roadmap_courses = db.Roadmap_courses.Find(id);
            if (roadmap_courses == null)
            {
                return HttpNotFound();
            }
            return View(roadmap_courses);
        }

        // GET: Admin/Roadmap_courses/Create
        public ActionResult Create()
        {
            ViewBag.RoadmapId = new SelectList(db.Roadmaps, "Id", "Name");
            return View();
        }

        // POST: Admin/Roadmap_courses/Create
        // To protect from overposting attacks, enable the specific properties you want to bind to, for 
        // more details see https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "Id,Name,Link,Order,RoadmapId")] Roadmap_courses roadmap_courses)
        {
            if (ModelState.IsValid)
            {
                db.Roadmap_courses.Add(roadmap_courses);
                db.SaveChanges();
                return RedirectToAction("Index");
            }

            ViewBag.RoadmapId = new SelectList(db.Roadmaps, "Id", "Name", roadmap_courses.RoadmapId);
            return View(roadmap_courses);
        }

        // GET: Admin/Roadmap_courses/Edit/5
        public ActionResult Edit(int? id)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            Roadmap_courses roadmap_courses = db.Roadmap_courses.Find(id);
            if (roadmap_courses == null)
            {
                return HttpNotFound();
            }
            ViewBag.RoadmapId = new SelectList(db.Roadmaps, "Id", "Name", roadmap_courses.RoadmapId);
            return View(roadmap_courses);
        }

        // POST: Admin/Roadmap_courses/Edit/5
        // To protect from overposting attacks, enable the specific properties you want to bind to, for 
        // more details see https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "Id,Name,Link,Order,RoadmapId")] Roadmap_courses roadmap_courses)
        {
            if (ModelState.IsValid)
            {
                db.Entry(roadmap_courses).State = EntityState.Modified;
                db.SaveChanges();
                return RedirectToAction("Index");
            }
            ViewBag.RoadmapId = new SelectList(db.Roadmaps, "Id", "Name", roadmap_courses.RoadmapId);
            return View(roadmap_courses);
        }

        // GET: Admin/Roadmap_courses/Delete/5
        public ActionResult Delete(int? id)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            Roadmap_courses roadmap_courses = db.Roadmap_courses.Find(id);
            if (roadmap_courses == null)
            {
                return HttpNotFound();
            }
            return View(roadmap_courses);
        }

        // POST: Admin/Roadmap_courses/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(int id)
        {
            Roadmap_courses roadmap_courses = db.Roadmap_courses.Find(id);
            db.Roadmap_courses.Remove(roadmap_courses);
            db.SaveChanges();
            return RedirectToAction("Index");
        }

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