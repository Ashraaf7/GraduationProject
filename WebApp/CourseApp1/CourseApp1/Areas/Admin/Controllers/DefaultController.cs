using CourseApp1.data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace CourseApp1.Areas.Admin.Controllers
{
    [Authorize(Roles = "Admin")]
    public class DefaultController : Controller
    {
        private courses_dbEntities db = new courses_dbEntities();
        // GET: Admin/Default
        public ActionResult Index()
        {
            var courses = db.Courses.ToList();
            var courses_count=db.Courses.Count();
            var categories_count=db.Categories.Count();
            var roadmaps_count=db.Roadmaps.Count();
            var rooms_count=db.Rooms.Count();
            ViewBag.Rooms = rooms_count;
            ViewBag.Roadmaps = roadmaps_count;
            ViewBag.Category = categories_count;
            ViewBag.Courses = courses_count;
            return View(courses);
        }
    }
}